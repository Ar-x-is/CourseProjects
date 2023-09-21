void charge_sym_hist_2(){

    TFile *file = new TFile("hist1.root");
    //TFile* outagain = new TFile("outagain.root", "RECREATE");
    TTree *tree020 = (TTree*)file->Get("pytree020");
    Int_t entries020 = tree020->GetEntries();

    const Int_t maxTrack = 10000;

    Int_t ntrack ;
    Double_t pid[maxTrack];

    tree020->SetBranchAddress("pid", &pid);
    tree020->SetBranchAddress("ntrack", &ntrack);

    TH1D *hist_charge_dist = new TH1D("hist_charge_dist", "Charge Asymmetry in the 0-20 Multiplicity Class", 100, -40, 40);

    for(Int_t ii=0; ii<entries020; ii++)  {
        tree020->GetEntry(ii);        
	    Int_t netcharge = 0;

        for(int i=0; i<ntrack; i++)  {
            if(pid[i]>0){
                netcharge += 1;
            }
            else if (pid[i]==0)
            {
                //cout<<"zero"<<endl;
                //netcharge -= 1;
            }
            else{
                netcharge-=1;
            }
         
        hist_charge_dist->Fill(netcharge);  
        }
    }
    //TF1* f1 = new TF1("f1", "10000000 * ROOT::Math::normal_pdf(x, 0, 10)", -45, 45);
    //f1->SetParameter(2, 7);
    TFile* outFile = new TFile("output.root", "UPDATE");
    //hist_charge_dist->Fit(f1, "", "", - 45, 45);
    hist_charge_dist->Fit("gaus");
    hist_charge_dist->Draw();
    delete outFile;
    //delete outagain;
}