void charge_sym_hist_2(){

    TCanvas *c1 = new TCanvas("c1","c1",500,500);
    c1->SetGrid();
   
    TFile *file = new TFile("hist1.root");
    TTree *tree020 = (TTree*)file->Get("pytree80100;5");
    Int_t entries020 = tree020->GetEntries();

    const Int_t maxTrack = 10000;

    Int_t ntrack ;
    Double_t pid[maxTrack];
    Double_t pT[maxTrack];
    Double_t eta[maxTrack];
    Double_t rap[maxTrack];
    Double_t phi[maxTrack];
    

    tree020->SetBranchAddress("pid", &pid);
    tree020->SetBranchAddress("ntrack", &ntrack);

    tree020->SetBranchAddress("phi",&phi);
    tree020->SetBranchAddress("pT",&pT);
    tree020->SetBranchAddress("eta",&eta);
    tree020->SetBranchAddress("rap", &rap);

    TH1D *hist_charge_dist = new TH1D("hist_charge_dist", "Charge Distribution", 100, -200, 200);

    TH1D *hmults = new TH1D("hmult","multiplicity",100,0.0,30.0);
    TH1D *hphis = new TH1D("hphi","Phi",100,-4.0,4.0);
    TH1D* hraps = new TH1D("hrap", "Rapidity", 100, -3.0, 3.0);
    TH1D* hetas = new TH1D("heta", "Eta", 100, -4.0, 4.0);
    TH1F* hpTs = new TH1F("hpT", "Transverse Momentum", 100, 0.2, 3);
    
    TH1D *hmulta = new TH1D("hmult","multiplicity",100,0.0,30.0);
    TH1D *hphia = new TH1D("hphi","Phi",100,-4.0,4.0);
    TH1D* hrapa = new TH1D("hrap", "Rapidity", 100, -3.0, 3.0);
    TH1D* hetaa = new TH1D("heta", "Eta", 100, -4.0, 4.0);
    TH1F* hpTa = new TH1F("hpT", "Transverse Momentum", 100, 0.2, 3);


    for(Int_t ii=0; ii<entries020; ii++)  {
        tree020->GetEntry(ii);        
	    Int_t netcharge = 0;

        for(int i=0; i<ntrack; i++)  {
            if(pid[i]>0){
                netcharge += 1;
            }
           
            else{
                netcharge-=1;
                
            }
         
        hist_charge_dist->Fill(netcharge);  

        if(netcharge<10 && netcharge>-10){

            for(int j=0; j<ntrack; j++) { //track loop is here


            Double_t eta1 = eta[j];
            Double_t phi1 = phi[j];
            Double_t rap1 = rap[j];
            Float_t pT1 = pT[j];

            hetas->Fill(eta1);
            hphis->Fill(phi1);
            hpTs->Fill(pT1);
            hraps->Fill(rap1);
            hmults->Fill(ntrack);
        }

        }
        else {

            for(int j=0; j<ntrack; j++) { //track loop is here
            
            Double_t eta1 = eta[j];
            Double_t phi1 = phi[j];
            Double_t rap1 = rap[j];
            Float_t pT1 = pT[j];

            hetaa->Fill(eta1);
            hphia->Fill(phi1);
            hpTa->Fill(pT1);
            hrapa->Fill(rap1);
            hmulta->Fill(ntrack);
        }

          
        }
    
    }
    }
    
    TFile* outFile = new TFile("output.root", "RECREATE");
    
  hist_charge_dist->Write();
  //hetas->Write();
  //hphis->Write();
  //hraps->Write()
  //hetaa->Write();
  //hphia->Write();
  //hrapa->Write();
  //hpTa->Write();
  //c1->Write();
  delete outFile;
    hpTs->Draw();
    hpTa->Draw("same");
    //hphis->Draw();
    //hphia->Draw("same")
    //hetas->Draw();
    //hetaa->Draw("same")
    //hmults->Draw();
    //hmulta->Draw("same")
    //hraps->Draw();
    //hrapa->Draw("same")
    //hist_charge_dist->Draw();
    delete hpTs;
    delete hpTa;
    delete hphis;
    delete hetaa;
    delete hphia;
    delete hetas;
    delete hmults;
    delete hmulta;
    delete hraps;
    delete hrapa;

}
